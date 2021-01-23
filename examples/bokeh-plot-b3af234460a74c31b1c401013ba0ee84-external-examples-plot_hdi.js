(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("288a5d7a-c80c-4f62-bc0c-1adab3234079");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '288a5d7a-c80c-4f62-bc0c-1adab3234079' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"5214c109-4329-4cba-90e3-98095c68ef8b":{"roots":{"references":[{"attributes":{},"id":"18387","type":"DataRange1d"},{"attributes":{},"id":"18439","type":"Selection"},{"attributes":{"overlay":{"id":"18410"}},"id":"18405","type":"LassoSelectTool"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"18439"},"selection_policy":{"id":"18438"}},"id":"18420","type":"ColumnDataSource"},{"attributes":{},"id":"18394","type":"BasicTicker"},{"attributes":{},"id":"18440","type":"UnionRenderers"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"18401"},{"id":"18402"},{"id":"18403"},{"id":"18404"},{"id":"18405"},{"id":"18406"},{"id":"18407"},{"id":"18408"}]},"id":"18411","type":"Toolbar"},{"attributes":{},"id":"18441","type":"Selection"},{"attributes":{},"id":"18391","type":"LinearScale"},{"attributes":{"axis":{"id":"18397"},"dimension":1,"ticker":null},"id":"18400","type":"Grid"},{"attributes":{"data":{"x":{"__ndarray__":"Ozjur1CtAsC23ktO02EBwK3Km1JpcwDANOEZUWfG/78vsQEAZEP9v3ZdLFZGXve/VdTyBqda9r8C2RQJVnv1v9+32mwBX/W/2tn/DTZx9L/j0M2zu1Tyv2RsFkbWO/K/UpmcutUM8r+qwNC7od7xv0dgj0bTcfG/I8EvqHBi8b9UbXjtSCfxvyn/WO8xSPC/I1I5YL977r8rkoDAKGXuvxmI2dx3Ae2/4WrbsswY7L9Yx6QeaAHsv8Gyps6Zr+q/b38ZZ2hB6r+TJ6uP6cDmv1l+btEBWua/3hrq+DXa5b9fElwFFVjkv9PBgXUJhuO/LzAEXv/W4r+fTmfMQZLivyCGTUgfueG/HVZYm70M4L/MOBTuhfffvz72iF3EaN6/GzsJv9C92r868Dj8pFnYv4Ysu/NPrNe/YFsKv7GG0r8HBqZ+8tDRv5NXky0EIMi/QAQJqbLnxL//8fEdHaHEv0rybMFHJ7q/9i7C/pEDt7/Yd6Yc2HW0vw6Xkue4w7C/Qv1I/SJ/pz+yU0ZVAZerPyM8x11kM64/si70m3WUrj/o5LgdMKHDP1MXORMz6sM/VHDCPh5xxD+oIbAtjS/JP4yBDGhMqso/1FJ98ePnyj99fDsFddrRP4XGdm2/c9M/Iynm06ng1D/iesMGZLPVP94q/Sy9Ztk/fe0OGTCf2z9LWPNdiu/cP74d5TMvSt0/EHU3QtHB3T8Adl0hK0zgP6bKedm9iuE/BnV+VGW04T+2JjkDi7bhP4PU/Bfs2+E/I/5asrkb4j/yeOtWqyHiPx141XQFFeM/+X/77Bct4z/1yl5HIpfjPwbzaScBuOM/rDQEfguL5D+YJorHb0nlP2aQf0FkruU/Q1lKFwn35T/15zqN5k3mP3SZeSnEyeY/ym18Gf666D/lXp9Qmt/oPzQEI3pNzOk/2YYZ3KDr6j95vzA9AfLtP1/jqJZiWO4/+9ik/Cr08z9J49njNZ/3P45DhwUb6Pc/1YosBTgO+D9n5jE5J078P4zF+a52NP0/B6FZZdej/T/BQyPQOf39PySDbqFefAFA1zb4y4e8BEA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"io8joF6l6j+UQmhjWTztP6ZqyFotGe8/Zg9zV8wc8D9oJ///TV7xP0XR6dTcUPQ/1pWGfKzS9D9/k3X7VEL1PxCkkkl/UPU/ExMA+WTH9T+OFxkmotX2P87J9NwU4vY/V7OxIpX59j+rnxcirxD3P9xPuFwWR/c/bh/oq8dO9z9WyUOJW2z3P2yAUwjn2/c/d6vxJxBh+D9129/PtWb4P/qdyQiiv/g/SCVJ08z5+D8qzlb4pf/4P1BTVowZVPk/JKA55qVv+T8bNhWcxU/6P2pgpIt/afo/SHnFgXKJ+j9o+6i++un6P4uPn6J9Hvs/9PN+KEBK+z9YLOaMb1v7P3ie7C24kfs/eeopmdD8+z/meD1CDwH8PzjhTnTnMvw/ndge6EWo/D/54Xhgy/T8P2+aiAF2Cv0/lLQeyCmv/T8/Pyuw4cX9P4fKJr3/ff4/vG9v1YSx/j/g4CAu7rX+P26Y9MHFLv8/iO4JcONH/z9BzBo/UVz/P0hrwzjief8/+5H6Rf4uAECnjKoCLjcAQHiOu8hmPABAXeg36yg9AEAnx+2ACZ0AQLvImZhRnwBAgxP28YijAEANgW1pfMkAQAxkQGNS1QBAl+qLHz/XAEDIt1NQpx0BQGhs1/Y7NwFAkmI+nQpOAUCuN2xANlsBQK7Sz9JrlgFA2O6QAfO5AUCFNd+l+M4BQNxRPvOi1AFAUXcjFB3cAUDAritkhQkCQFU5L7tXMQJAoc6Pqow2AkDXJGdg0TYCQJCa/4J9OwJAxF9LNndDAkAeb91qNUQCQASvmq6gYgJA/2+f/aJlAkBf2etI5HICQGE+7SQAdwJAlobAb2GRAkDTRPH4LakCQA3yL4jMtQJAKEvpIuG+AkD/XKfRvMkCQC4zL4U42QJAuY0vw18XA0Dd6xNK8xsDQIZgRK+JOQNA2zCDG3RdA0DvF6YnQL4DQGwc1VIMywNAPzYpvwr9BEDSePZ4zecFQOTQYcEG+gVAtSJLAY4DBkCaeUzOiRMHQGNxvqsdTQdAQmhW2fVoB0Dw0Ah0Tn8HQJJBt1AvvghAbBv85UNeCkA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"18441"},"selection_policy":{"id":"18440"}},"id":"18425","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"18425"},"glyph":{"id":"18426"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"18427"},"selection_glyph":null,"view":{"id":"18429"}},"id":"18428","type":"GlyphRenderer"},{"attributes":{},"id":"18406","type":"UndoTool"},{"attributes":{},"id":"18385","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"18410","type":"PolyAnnotation"},{"attributes":{},"id":"18407","type":"SaveTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"18421","type":"Patch"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"18426","type":"Line"},{"attributes":{},"id":"18433","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"18420"}},"id":"18424","type":"CDSView"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"18422","type":"Patch"},{"attributes":{"source":{"id":"18425"}},"id":"18429","type":"CDSView"},{"attributes":{"callback":null},"id":"18408","type":"HoverTool"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"18427","type":"Line"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"18409","type":"BoxAnnotation"},{"attributes":{},"id":"18438","type":"UnionRenderers"},{"attributes":{"text":""},"id":"18431","type":"Title"},{"attributes":{},"id":"18402","type":"PanTool"},{"attributes":{"overlay":{"id":"18409"}},"id":"18403","type":"BoxZoomTool"},{"attributes":{},"id":"18389","type":"LinearScale"},{"attributes":{"formatter":{"id":"18433"},"ticker":{"id":"18394"}},"id":"18393","type":"LinearAxis"},{"attributes":{},"id":"18404","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"18420"},"glyph":{"id":"18421"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"18422"},"selection_glyph":null,"view":{"id":"18424"}},"id":"18423","type":"GlyphRenderer"},{"attributes":{"formatter":{"id":"18435"},"ticker":{"id":"18398"}},"id":"18397","type":"LinearAxis"},{"attributes":{"axis":{"id":"18393"},"ticker":null},"id":"18396","type":"Grid"},{"attributes":{},"id":"18398","type":"BasicTicker"},{"attributes":{},"id":"18435","type":"BasicTickFormatter"},{"attributes":{"below":[{"id":"18393"}],"center":[{"id":"18396"},{"id":"18400"}],"left":[{"id":"18397"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"18423"},{"id":"18428"}],"title":{"id":"18431"},"toolbar":{"id":"18411"},"toolbar_location":"above","x_range":{"id":"18385"},"x_scale":{"id":"18389"},"y_range":{"id":"18387"},"y_scale":{"id":"18391"}},"id":"18384","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"18401","type":"ResetTool"}],"root_ids":["18384"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"5214c109-4329-4cba-90e3-98095c68ef8b","root_ids":["18384"],"roots":{"18384":"288a5d7a-c80c-4f62-bc0c-1adab3234079"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();