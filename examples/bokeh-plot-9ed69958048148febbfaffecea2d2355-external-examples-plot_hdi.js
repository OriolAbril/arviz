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
    
      
      
    
      var element = document.getElementById("1193818d-e485-470c-a18f-ae2286b43936");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '1193818d-e485-470c-a18f-ae2286b43936' but no matching script tag was found.")
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
                    
                  var docs_json = '{"65abfdbd-bc7e-43c8-898f-970a5168f883":{"roots":{"references":[{"attributes":{"data_source":{"id":"5272"},"glyph":{"id":"5273"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5274"},"selection_glyph":null,"view":{"id":"5276"}},"id":"5275","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"5261"}},"id":"5255","type":"BoxZoomTool"},{"attributes":{},"id":"5284","type":"BasicTickFormatter"},{"attributes":{},"id":"5243","type":"LinearScale"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5253"},{"id":"5254"},{"id":"5255"},{"id":"5256"},{"id":"5257"},{"id":"5258"},{"id":"5259"},{"id":"5260"}]},"id":"5263","type":"Toolbar"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5278","type":"Line"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5262","type":"PolyAnnotation"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5288"},"selection_policy":{"id":"5289"}},"id":"5272","type":"ColumnDataSource"},{"attributes":{"source":{"id":"5272"}},"id":"5276","type":"CDSView"},{"attributes":{"data_source":{"id":"5277"},"glyph":{"id":"5278"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5279"},"selection_glyph":null,"view":{"id":"5281"}},"id":"5280","type":"GlyphRenderer"},{"attributes":{},"id":"5250","type":"BasicTicker"},{"attributes":{"data":{"x":{"__ndarray__":"z6WzwdOMAcAwbUyeX7EAwF6dReXScvy/NYrAjFS2+79BRnaRJ8P6v06TDhPTLPi/QUedgoqx97+q3F5VGQP3vyn3Xr41vva/Mtfe2O1Z9b+c6NDOmy30v3Pn2YOtNvO/LNJ98obA8b8pUdMJ04/xv+R20K85kfC/wXcF7v9377/3FUhI/n3tv0ZN5KW0ROy/rWSzkEtw67+QlJuJIrTnv0mi/cm+h+e//meFvwtO579Rqks4MZ3lv2UM0Z/xTuW/OOA2xXOj5L+HX0YDE4jjv39UpHKq5eK/uqfo0EME4r86tlhHLc3ev/yZaXqkUd2/7rS9wnzL3L/qDHsOI8/av/S3495EQ82/rI4Y+O87xL+RSX4nOafBvyOu76SUpcG/jqo1JbKjv7+91gG2Kmy4v+Sl4V2hX6i/WQxsiWvupr8INRCdVtGWv5rFcCfZBom/DiV2HgYrar8rnHs4nycxv1Y7GiE5CXQ/dzIvCubWdj9Pl056bqybP8hj4bykMLA/kOB1fqQqvD/QTZIUfg6+P4D2MmMVMsE/CMh7HidnyD+vZOIYZMTIP/wWnVac9cg/IVcKd0S2yT9mra0RJfLNPzn6pb1zo9A/n8/zm1sf0T8DJ9XI6DzRP50x+dtLtNE/GDhVJWs80z9iAlcY/T3TP2A6zu9VltQ/TfpGICqi1T/L3NZydOnVP+yJk66M7dg/Yd0A+1fA2T/bEQyZoizbPyGzBWDE9d0/pnKuMDsg3z+i18FCgCDfPyowGWcbvd8/DxEhF8Io4D/9Qux5U9rgP+i7egSkiuE/vdrSwv/94j+fkN+NZc/kP+QmO4ZARuk/UKrSZ8uv6T87mMOrndTpPxx/iCiE2ek/L+w89XYL6j/uzPPxXGfqP+TMp2MP8+o/i6NiwfOO6z96yRG6liLsPyxj08xty+w/aKMUXx767j+DTypiX0DvP97vmEwi0PA/MZA4gRtq8T966wUBFX71Px1ZbXY2L/c/IGsJrNSf9z9CCAu3ceH5PxLlw2pF8vk/y5Ntsmnw/D+E0zMwc2MBQMd4NBXolwJAU136ekqMA0A=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"YrSYfFjm7D+gJWfDQJ3uP1ExXY2WxvE/5rqfudUk8j/g3EQ3bJ7yP1m2eHaW6fM/YFyxvjon9D+rkVBVc370P2yE0CDloPQ/Z5SQEwlT9T+yi5cYMun1P0YMEz6pZPY/6hbBhrwf9z9sVxZ7Fjj3P47EFyhjt/c/EKJ+BAAi+D+C+u1tgKD4P67shtbS7vg/1SbTG+0j+T/cGpld9xL6P26XgE0QHvo/AKYeEH0s+j9sFe2xs5j6P+e8C5hDrPo/8keyDiPX+j8eaC4/+x37P+DqVmOVRvs/EtbFC+9++z856RRXWib8P8DMsnDLVfw/YkmoZ5Bm/D9jnjCeG6b8P4HEEbLLK/4/FXd+AEG8/j9nG4htjOX+Px4FsbWm5f4/rFLWbuIC/z9K8U+qnjz/P2h5iHqBnv8/z0/aUUak/z+W38VSXdL/PzqP2Cb55v8/d2J4PnX5/z8jPAbDdv//P49GSE4CBQBAzYuCubUFAECXTnpurBsAQI+F85LCQABAgtf5kapwAEA3SVL4OXgAQLSXGauQiQBAQN7zODnDAEAlE8cgI8YAQLjotOKsxwBAuVK4I7LNAEBrbY0oke8AQKRf2js3CgFA+jy/ufURAUBwUo2MzhMBQBqTv71EGwFAglNVssYzAUAmcIXR3zMBQKbj/F5lSQFApW8EoiJaAUDNbS1Hl14BQJ846crYjgFA1g2wfwWcAUAewZApyrIBQDJbAEZc3wFAKucKswPyAUB6HSwECPIBQAOTcbbR+wFAIiLkQhgFAkBgiD1vShsCQH1Xj4BUMQJAWFta+L9fAkAU8rux7JkCQNxkxxDIKANASlX6bPk1A0AHc3i1kzoDQOQPEYUwOwNAhp2n3m5BA0CeeT6e60wDQJz5dOxhXgNAcVQseN5xA0AvOULXUoQDQGZsmrltmQNAbZTiy0PfA0DwSUXsC+gDQPg7JpMINARADCRO4IZaBEDeekFAhV8FQEdWm53NywVAyFoCK/XnBUAQwsJtXHgGQET5sFqRfAZA82SbbBo8B0DC6RmYubEIQGQ8mgr0SwlAqi59PSXGCUA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5290"},"selection_policy":{"id":"5291"}},"id":"5277","type":"ColumnDataSource"},{"attributes":{},"id":"5246","type":"BasicTicker"},{"attributes":{},"id":"5254","type":"PanTool"},{"attributes":{"axis":{"id":"5249"},"dimension":1,"ticker":null},"id":"5252","type":"Grid"},{"attributes":{"text":""},"id":"5282","type":"Title"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5274","type":"Patch"},{"attributes":{"callback":null},"id":"5260","type":"HoverTool"},{"attributes":{},"id":"5241","type":"LinearScale"},{"attributes":{},"id":"5286","type":"BasicTickFormatter"},{"attributes":{"axis":{"id":"5245"},"ticker":null},"id":"5248","type":"Grid"},{"attributes":{},"id":"5288","type":"Selection"},{"attributes":{"below":[{"id":"5245"}],"center":[{"id":"5248"},{"id":"5252"}],"left":[{"id":"5249"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5275"},{"id":"5280"}],"title":{"id":"5282"},"toolbar":{"id":"5263"},"toolbar_location":"above","x_range":{"id":"5237"},"x_scale":{"id":"5241"},"y_range":{"id":"5239"},"y_scale":{"id":"5243"}},"id":"5236","subtype":"Figure","type":"Plot"},{"attributes":{"overlay":{"id":"5262"}},"id":"5257","type":"LassoSelectTool"},{"attributes":{},"id":"5289","type":"UnionRenderers"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5261","type":"BoxAnnotation"},{"attributes":{},"id":"5259","type":"SaveTool"},{"attributes":{"formatter":{"id":"5286"},"ticker":{"id":"5250"}},"id":"5249","type":"LinearAxis"},{"attributes":{},"id":"5258","type":"UndoTool"},{"attributes":{},"id":"5291","type":"UnionRenderers"},{"attributes":{},"id":"5237","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5273","type":"Patch"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5279","type":"Line"},{"attributes":{"formatter":{"id":"5284"},"ticker":{"id":"5246"}},"id":"5245","type":"LinearAxis"},{"attributes":{},"id":"5256","type":"WheelZoomTool"},{"attributes":{"source":{"id":"5277"}},"id":"5281","type":"CDSView"},{"attributes":{},"id":"5239","type":"DataRange1d"},{"attributes":{},"id":"5253","type":"ResetTool"},{"attributes":{},"id":"5290","type":"Selection"}],"root_ids":["5236"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"65abfdbd-bc7e-43c8-898f-970a5168f883","root_ids":["5236"],"roots":{"5236":"1193818d-e485-470c-a18f-ae2286b43936"}}];
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