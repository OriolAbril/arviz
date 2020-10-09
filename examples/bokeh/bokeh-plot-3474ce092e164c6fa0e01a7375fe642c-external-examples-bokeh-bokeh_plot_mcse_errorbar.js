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
    
      
      
    
      var element = document.getElementById("ac7b5533-231f-4105-80a3-cfed35ad9c13");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'ac7b5533-231f-4105-80a3-cfed35ad9c13' but no matching script tag was found.")
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
                    
                  var docs_json = '{"c694bde6-e792-4682-a6ef-71788910eb01":{"roots":{"references":[{"attributes":{},"id":"21984","type":"BasicTickFormatter"},{"attributes":{},"id":"21949","type":"BasicTicker"},{"attributes":{"axis_label":"Value $\\\\pm$ MCSE for quantiles","formatter":{"id":"21986"},"ticker":{"id":"21949"}},"id":"21948","type":"LinearAxis"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21972","type":"Dash"},{"attributes":{},"id":"21953","type":"PanTool"},{"attributes":{"overlay":{"id":"21960"}},"id":"21954","type":"BoxZoomTool"},{"attributes":{"overlay":{"id":"21961"}},"id":"21956","type":"LassoSelectTool"},{"attributes":{},"id":"21952","type":"ResetTool"},{"attributes":{},"id":"21958","type":"SaveTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21952"},{"id":"21953"},{"id":"21954"},{"id":"21955"},{"id":"21956"},{"id":"21957"},{"id":"21958"},{"id":"21959"}]},"id":"21962","type":"Toolbar"},{"attributes":{"children":[[{"id":"21935"},0,0]]},"id":"21995","type":"GridBox"},{"attributes":{"children":[{"id":"21997"},{"id":"21995"}]},"id":"21998","type":"Column"},{"attributes":{"axis":{"id":"21948"},"dimension":1,"ticker":null},"id":"21951","type":"Grid"},{"attributes":{},"id":"21992","type":"Selection"},{"attributes":{},"id":"21938","type":"DataRange1d"},{"attributes":{},"id":"21942","type":"LinearScale"},{"attributes":{"data":{"xs":[[0.05,0.05],[0.09736842105263158,0.09736842105263158],[0.14473684210526316,0.14473684210526316],[0.19210526315789472,0.19210526315789472],[0.23947368421052628,0.23947368421052628],[0.28684210526315784,0.28684210526315784],[0.33421052631578946,0.33421052631578946],[0.381578947368421,0.381578947368421],[0.4289473684210526,0.4289473684210526],[0.47631578947368414,0.47631578947368414],[0.5236842105263158,0.5236842105263158],[0.5710526315789474,0.5710526315789474],[0.618421052631579,0.618421052631579],[0.6657894736842105,0.6657894736842105],[0.7131578947368421,0.7131578947368421],[0.7605263157894736,0.7605263157894736],[0.8078947368421052,0.8078947368421052],[0.8552631578947368,0.8552631578947368],[0.9026315789473683,0.9026315789473683],[0.95,0.95]],"ys":[[0.06542507260989384,0.07779740932825195],[0.08783828101486021,0.09606135597316004],[0.09938050528645676,0.10552315572600299],[0.10794861576931342,0.11348270780810846],[0.11602734321885094,0.12203033330056018],[0.12350211909750375,0.12787167939093885],[0.12891616867169003,0.13323376159544686],[0.13473004088066556,0.13958486303584752],[0.14096481555257187,0.1460531576744907],[0.14714684275886325,0.15141900007039433],[0.15249024001585332,0.15717175454660753],[0.15761692093229293,0.16322637365277673],[0.16410729163047846,0.16867037944462424],[0.17025660267247006,0.17489182297617661],[0.17657098829958856,0.18216819265595885],[0.185369502560243,0.18948857007780892],[0.19264880841245133,0.19711830100588604],[0.20027922695307082,0.2044362824100993],[0.2124901683329975,0.2184324622519046],[0.23108929611368745,0.2369230522137875]]},"selected":{"id":"21992"},"selection_policy":{"id":"21993"}},"id":"21976","type":"ColumnDataSource"},{"attributes":{},"id":"21991","type":"UnionRenderers"},{"attributes":{"source":{"id":"21971"}},"id":"21975","type":"CDSView"},{"attributes":{"data_source":{"id":"21976"},"glyph":{"id":"21977"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21978"},"selection_glyph":null,"view":{"id":"21980"}},"id":"21979","type":"GlyphRenderer"},{"attributes":{},"id":"21986","type":"BasicTickFormatter"},{"attributes":{},"id":"21936","type":"DataRange1d"},{"attributes":{"line_color":{"value":"#1f77b4"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"21977","type":"MultiLine"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21960","type":"BoxAnnotation"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"+Pb8QR1Vsj+MDk73BYq3PyGJcH1IOro/R5qJktxXvD8QN5uHrHi+P+EQ5SCCFsA/nVhJgBDHwD+XHUgZYI7BP8j2F6KAXsI/quCp6LMbwz/9bt2KgNHDP6iPWFCyiMQ/uNKGtzpMxT8aURts6RbGP6Hdfy6V9cY/RM+ntqz9xz+OhgC/8fHIP+aFOeHb5sk//PpdgzyUyz/cM4wQ6vPNPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"21990"},"selection_policy":{"id":"21991"}},"id":"21971","type":"ColumnDataSource"},{"attributes":{"toolbars":[{"id":"21962"}],"tools":[{"id":"21952"},{"id":"21953"},{"id":"21954"},{"id":"21955"},{"id":"21956"},{"id":"21957"},{"id":"21958"},{"id":"21959"}]},"id":"21996","type":"ProxyToolbar"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"21978","type":"MultiLine"},{"attributes":{},"id":"21993","type":"UnionRenderers"},{"attributes":{"below":[{"id":"21944"}],"center":[{"id":"21947"},{"id":"21951"}],"left":[{"id":"21948"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21974"},{"id":"21979"}],"title":{"id":"21981"},"toolbar":{"id":"21962"},"toolbar_location":null,"x_range":{"id":"21936"},"x_scale":{"id":"21940"},"y_range":{"id":"21938"},"y_scale":{"id":"21942"}},"id":"21935","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"21976"}},"id":"21980","type":"CDSView"},{"attributes":{"text":"sigma_a"},"id":"21981","type":"Title"},{"attributes":{},"id":"21955","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"21971"},"glyph":{"id":"21972"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21973"},"selection_glyph":null,"view":{"id":"21975"}},"id":"21974","type":"GlyphRenderer"},{"attributes":{},"id":"21957","type":"UndoTool"},{"attributes":{"toolbar":{"id":"21996"},"toolbar_location":"above"},"id":"21997","type":"ToolbarBox"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21961","type":"PolyAnnotation"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21973","type":"Dash"},{"attributes":{},"id":"21990","type":"Selection"},{"attributes":{"axis":{"id":"21944"},"ticker":null},"id":"21947","type":"Grid"},{"attributes":{},"id":"21940","type":"LinearScale"},{"attributes":{"callback":null},"id":"21959","type":"HoverTool"},{"attributes":{},"id":"21945","type":"BasicTicker"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"21984"},"ticker":{"id":"21945"}},"id":"21944","type":"LinearAxis"}],"root_ids":["21998"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"c694bde6-e792-4682-a6ef-71788910eb01","root_ids":["21998"],"roots":{"21998":"ac7b5533-231f-4105-80a3-cfed35ad9c13"}}];
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