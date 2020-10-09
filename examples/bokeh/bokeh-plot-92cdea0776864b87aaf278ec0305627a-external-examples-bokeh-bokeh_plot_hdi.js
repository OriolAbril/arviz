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
    
      
      
    
      var element = document.getElementById("67d8189c-0de7-4700-bdf5-ca0171b2b9ee");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '67d8189c-0de7-4700-bdf5-ca0171b2b9ee' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
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
                    
                  var docs_json = '{"065e64d4-249f-4da8-851a-e45f6014a22c":{"roots":{"references":[{"attributes":{"text":""},"id":"5282","type":"Title"},{"attributes":{},"id":"5250","type":"BasicTicker"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5288"},"selection_policy":{"id":"5289"}},"id":"5272","type":"ColumnDataSource"},{"attributes":{"data":{"x":{"__ndarray__":"HOhvQbuHAsDRyiFnD9b8v37txNmARfu/aL735UXO+b994YlVvH/4v6/tl0J8Mve/cR4E2LzO9r+mlogSI832v0TBvpDsOva/MwxNKb3i9L/xYNb6PkHxv6nNtCumQPC/ZxiGgegh8L+/g3sL6g/wv+e03D6f2e+/w0qsY4sj7781rz1oGEPuv9REkyXaB+6/eK67WR0B7b/O4QkdfmTsv9raX66pRey/3nymr4FA679Se4rn1TDqv04ACuOzMOm/60BujQlW6L/1aIlov5fjvyoq2fXMj+O/aLOImtBu4b/w5DzKxUTgvyxUvFCJXdy/Ixu/RuMa3L92kWAAOZHav0Pih8DB6Nm/oWV9YWJq2b8t8DE81h7Yv8iVEsXb5NW/Y1081oj91L/YV27eR6LTv4He1AseGtO/Q1AMS9Kx0r+CKALR3wTSv3iLsHINFsq/NBkFky5SyL8hrtlKXyXHv7DrhYKTH8e/+YLTpka6wr+Aa849FrHCv9SKDgGxGL2/GCfOZqcut78A3/vpTQG3v33zBcfyL5G/jo7Tbto3kL+cAC7W7X+EP/j/D32uBKM/Gm1yCnj3xT+Kc1KUyazPP9eZvGFqjNc/ENz0xHwJ2D8un7QQ+g3eP69DiOlp9t4/fczEmkLc4T8d3Byw/+nhP/Dvc5LL8uE/0RhcXs+54z/2qocPUoDkP7tGYlHDxeQ/U2QLzUk75T9juoVB4E3lPw7FZrM5U+U/ExZB9IKM5T8TblnEn5blPzwfWHL0tOU/fd11cgR46D8djOrkLvvoP904R0YiU+k/RjBoMN7A6j/25Wp6+JHrP2UfLjJkpOs/HPJmzi057D8EdJ5TJkPsP19hDsqEw+0/KGnKiQ9J7z/Z93TN1VTwP3vHPrq9c/A/XNiGHHqi8D/GzH9/OEbxP/Li2LDWS/E/iQ7GjcgT8j86Xu1sbI3yPwU9Ipb8UvQ/zjj+6ONq9T+ZNGgfjzv3P7Egwta1Q/g/gI6tptDK+j8RX26h3/T6P2teCIbscv0/JBCdcGWH/T8ULrfDwd//P7iivoT1ggNAl77sapMNBkA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"yC8gfYnw6j+YGm9M+JTxP0GJHZM/XfI/zCAEDd0Y8z9CDzvVIcDzPygJtN7BZvQ/yPD9k6GY9D+ttLt2bpn0P16foLeJ4vQ/5nlZa6GO9T+Iz5SCYF/3PyyZJeqs3/c/zPM8vwvv9z8gPkL6Cvj3P8bSSDCYCfg/T+0UJx03+D8zlPDlOW/4P8sum3YJfvg/YhSRqbi/+D+Mh7144Ob4P0oJaJSV7vg/yGAWlN8v+T8sYR2GynP5P+x/PQfTs/k/xW+knH3q+T/Dpd0lEBr7P3a1icIMHPs/JtNd2Uuk+z/ExnCNzu77P3p16NVOdPw/nBwol6N8/D/R7fPf2K38P7gD78fnwvw/TFPQs7PS/D/6wXk4Jfz8P0etXYdkQ/0/VHQ45U5g/T8FNTIEt4v9PzBkhT68nP0/+HWetsWp/T/wut8FZL/9P0j31CifXv4/ba7PFt16/j8eZVILqo3+P0Wh18cGjv4/0MeSlVvU/j9IGSOc7tT+P6mL93c6F/8/x47JxIpG/z8IIbCQ9Uf/Pxn0cRqg3f8/41giS5Df/z8AF+v2PwoAQAAg+lwJJgBAaZNTwLuvAECck6JMZv0AQJ3JG6bGeAFAwU1PzJeAAUDzSQuh3+ABQDuEmJ5m7wFAkJlYU4g7AkCEmwP2Pz0CQP59TnJZPgJAGoPL6zl3AkBf9fBBCpACQNdILGq4mAJAimyhOWmnAkBMtzAIvKkCQKLYbDZnqgJAwiKIXpCxAkDCLYv407ICQOgDS46etgJAsLtOjgAPA0CEUZ3cZR8DQBznyEhkKgNACQYNxhtYA0C/XE0PP3IDQO3DRYaMdANARN7MuSWHA0CAznPKZIgDQCzMQZlwuANAJU058SHpA0D2PV1zNRUEQN+xj27vHARAF7Yhh54oBEAy898fjlEEQLw4Nqz1UgRAooNxI/KEBECOVzsbW6MEQEGPiCW/FAVANI4/+rhaBUAmDdrH484FQCyIsHXtEAZAoGOrKbSyBkDEl1voN70GQJsXgiG7XAdACUQnXNlhB0CFy+1w8PcHQFxRX8J6wQlATF92tckGC0A=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5290"},"selection_policy":{"id":"5291"}},"id":"5277","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"5277"},"glyph":{"id":"5278"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5279"},"selection_glyph":null,"view":{"id":"5281"}},"id":"5280","type":"GlyphRenderer"},{"attributes":{},"id":"5243","type":"LinearScale"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5262","type":"PolyAnnotation"},{"attributes":{},"id":"5254","type":"PanTool"},{"attributes":{},"id":"5256","type":"WheelZoomTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5253"},{"id":"5254"},{"id":"5255"},{"id":"5256"},{"id":"5257"},{"id":"5258"},{"id":"5259"},{"id":"5260"}]},"id":"5263","type":"Toolbar"},{"attributes":{"axis":{"id":"5245"},"ticker":null},"id":"5248","type":"Grid"},{"attributes":{"axis":{"id":"5249"},"dimension":1,"ticker":null},"id":"5252","type":"Grid"},{"attributes":{},"id":"5284","type":"BasicTickFormatter"},{"attributes":{},"id":"5246","type":"BasicTicker"},{"attributes":{},"id":"5239","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5273","type":"Patch"},{"attributes":{},"id":"5241","type":"LinearScale"},{"attributes":{"formatter":{"id":"5286"},"ticker":{"id":"5246"}},"id":"5245","type":"LinearAxis"},{"attributes":{"formatter":{"id":"5284"},"ticker":{"id":"5250"}},"id":"5249","type":"LinearAxis"},{"attributes":{"overlay":{"id":"5261"}},"id":"5255","type":"BoxZoomTool"},{"attributes":{},"id":"5290","type":"Selection"},{"attributes":{},"id":"5291","type":"UnionRenderers"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5261","type":"BoxAnnotation"},{"attributes":{"source":{"id":"5272"}},"id":"5276","type":"CDSView"},{"attributes":{},"id":"5288","type":"Selection"},{"attributes":{"callback":null},"id":"5260","type":"HoverTool"},{"attributes":{"below":[{"id":"5245"}],"center":[{"id":"5248"},{"id":"5252"}],"left":[{"id":"5249"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5275"},{"id":"5280"}],"title":{"id":"5282"},"toolbar":{"id":"5263"},"toolbar_location":"above","x_range":{"id":"5237"},"x_scale":{"id":"5241"},"y_range":{"id":"5239"},"y_scale":{"id":"5243"}},"id":"5236","subtype":"Figure","type":"Plot"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5278","type":"Line"},{"attributes":{"data_source":{"id":"5272"},"glyph":{"id":"5273"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5274"},"selection_glyph":null,"view":{"id":"5276"}},"id":"5275","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"5262"}},"id":"5257","type":"LassoSelectTool"},{"attributes":{},"id":"5253","type":"ResetTool"},{"attributes":{},"id":"5289","type":"UnionRenderers"},{"attributes":{},"id":"5258","type":"UndoTool"},{"attributes":{"source":{"id":"5277"}},"id":"5281","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5279","type":"Line"},{"attributes":{},"id":"5237","type":"DataRange1d"},{"attributes":{},"id":"5259","type":"SaveTool"},{"attributes":{},"id":"5286","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5274","type":"Patch"}],"root_ids":["5236"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"065e64d4-249f-4da8-851a-e45f6014a22c","root_ids":["5236"],"roots":{"5236":"67d8189c-0de7-4700-bdf5-ca0171b2b9ee"}}];
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